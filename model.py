import logging

from neuralgym.models import Model # pip install git+https://github.com/JiahuiYu/neuralgym
from neuralgym.ops.gan_ops import gan_wgan_loss, gradients_penalty
from neuralgym.ops.gan_ops import random_interpolates
from neuralgym.ops.layers import flatten, resize
from neuralgym.ops.summary_ops import gradients_summary
from neuralgym.ops.summary_ops import scalar_summary, images_summary
from tensorflow.contrib.framework.python.ops import arg_scope

from function import bbox2mask, local_patch, resize_z_like
from function import gen_conv, gen_deconv, dis_conv, e_conv, gen_conv_add_z,_residual_block
from function import resize_mask_like, contextual_attention
import tensorflow as tf

logger = logging.getLogger()


class PIIGANModel(Model):

    def __init__(self):
        super().__init__('PIIGANModel')

    def build_inpaint_net(self, x, mask, z, config = None, reuse = False,
                          training = True, padding = 'SAME', name = 'inpaint_net'):
        """Inpaint network.

        Args:
            x: incomplete image, [-1, 1]
            mask: mask region {0, 1}
        Returns:
            [-1, 1] as predicted image
        """
        xin = x
        offset_flow = None
        ones_x = tf.ones_like(x)[:, :, :, 0:1]
        z = z * mask
        z_shape_down1 = x.get_shape().as_list()[1] // 2
        z_feature_down1 = resize_z_like(z, [z_shape_down1, z_shape_down1])
        z_feature_down2 = resize_z_like(z, [z_shape_down1 // 2, z_shape_down1 // 2])
        x = tf.concat([x, z, ones_x, ones_x * mask], axis = 3)
        # two stage network
        cnum = 32
        with tf.variable_scope(name, reuse=reuse), \
             arg_scope([gen_conv, gen_deconv, gen_conv_add_z],
                       training=training, padding=padding):
            # stage1
            x = gen_conv_add_z(x, z, cnum, 5, 1, name='conv1')
            x = gen_conv_add_z(x, z_feature_down1, 2 * cnum, 3, 2, name='conv2_downsample')
            x = gen_conv_add_z(x, z_feature_down1, 2 * cnum, 3, 1, name='conv3')

            x = gen_conv_add_z(x, z_feature_down2, 4 * cnum, 3, 2, name='conv4_downsample')
            x = gen_conv_add_z(x, z_feature_down2, 4 * cnum, 3, 1, name='conv5')
            x = gen_conv_add_z(x, z_feature_down2, 4 * cnum, 3, 1, name='conv6')
            mask_s = resize_mask_like(mask, x)

            x = gen_conv_add_z(x, z_feature_down2, 4 * cnum, 3, rate=2, name='conv7_atrous')
            x = gen_conv_add_z(x, z_feature_down2, 4 * cnum, 3, rate=4, name='conv8_atrous')
            x = gen_conv_add_z(x, z_feature_down2, 4 * cnum, 3, rate=8, name='conv9_atrous')
            x = gen_conv_add_z(x, z_feature_down2, 4 * cnum, 3, rate=16, name='conv10_atrous')

            x = gen_conv_add_z(x, z_feature_down2, 4 * cnum, 3, 1, name='conv11')
            x = gen_conv_add_z(x, z_feature_down2, 4 * cnum, 3, 1, name='conv12')

            x = gen_deconv(x, 2 * cnum, name='conv13_upsample')
            x = gen_conv(x, 2 * cnum, 3, 1, name='conv14')

            x = gen_deconv(x, cnum, name='conv15_upsample')
            x = gen_conv(x, cnum // 2, 3, 1, name='conv16')
            x = gen_conv(x, 3, 3, 1, activation=None, name='conv17')
            x = tf.clip_by_value(x, -1., 1.)
            x_stage1 = x
            # return x_stage1, None, None

            # stage2, paste result as input
            # x = tf.stop_gradient(x)
            x = x * mask + xin * (1. - mask)
            x.set_shape(xin.get_shape().as_list())
            # conv branch
            xnow = tf.concat([x, z, ones_x, ones_x * mask], axis=3)
            x = gen_conv_add_z(xnow, z, cnum, 5, 1, name='xconv1')

            x = gen_conv_add_z(x, z_feature_down1, cnum, 3, 2, name='xconv2_downsample')
            x = gen_conv_add_z(x, z_feature_down1, 2 * cnum, 3, 1, name='xconv3')

            x = gen_conv_add_z(x, z_feature_down2, 2 * cnum, 3, 2, name='xconv4_downsample')
            x = gen_conv_add_z(x, z_feature_down2, 4 * cnum, 3, 1, name='xconv5')
            x = gen_conv_add_z(x, z_feature_down2, 4 * cnum, 3, 1, name='xconv6')

            x = gen_conv_add_z(x, z_feature_down2, 4 * cnum, 3, rate=2, name='xconv7_atrous')
            x = gen_conv_add_z(x, z_feature_down2, 4 * cnum, 3, rate=4, name='xconv8_atrous')
            x = gen_conv_add_z(x, z_feature_down2, 4 * cnum, 3, rate=8, name='xconv9_atrous')
            x = gen_conv_add_z(x, z_feature_down2, 4 * cnum, 3, rate=16, name='xconv10_atrous')
            x_hallu = x

            x = gen_conv_add_z(xnow, z, cnum, 5, 1, name='pmconv1')

            x = gen_conv_add_z(x, z_feature_down1, cnum, 3, 2, name='pmconv2_downsample')
            x = gen_conv_add_z(x, z_feature_down1, 2 * cnum, 3, 1, name='pmconv3')

            x = gen_conv_add_z(x, z_feature_down2, 4 * cnum, 3, 2, name='pmconv4_downsample')
            x = gen_conv_add_z(x, z_feature_down2, 4 * cnum, 3, 1, name='pmconv5')

            x = gen_conv_add_z(x, z_feature_down2, 4 * cnum, 3, 1, name='pmconv6', activation=tf.nn.relu)
            x, offset_flow = contextual_attention(x, x, mask_s, 3, 1, rate=2)
            x = gen_conv_add_z(x, z_feature_down2, 4 * cnum, 3, 1, name='pmconv9')
            x = gen_conv_add_z(x, z_feature_down2, 4 * cnum, 3, 1, name='pmconv10')
            pm = x
            x = tf.concat([x_hallu, pm], axis=3)

            x = gen_conv_add_z(x, z_feature_down2, 4 * cnum, 3, 1, name='allconv11')
            x = gen_conv_add_z(x, z_feature_down2, 4 * cnum, 3, 1, name='allconv12')
            x = gen_deconv(x, 2 * cnum, name='allconv13_upsample')
            x = gen_conv(x, 2 * cnum, 3, 1, name='allconv14')
            x = gen_deconv(x, cnum, name='allconv15_upsample')
            x = gen_conv(x, cnum // 2, 3, 1, name='allconv16')
            x = gen_conv(x, 3, 3, 1, activation=None, name='allconv17')
            x_stage2 = tf.clip_by_value(x, -1., 1.)
        return x_stage1, x_stage2, offset_flow

    def build_wgan_local_discriminator(self, x, reuse = False, training = True):
        with tf.variable_scope('discriminator_local', reuse = reuse):
            cnum = 64
            x = dis_conv(x, cnum, name = 'conv1', training = training)
            x = dis_conv(x, cnum * 2, name = 'conv2', training = training)
            x = dis_conv(x, cnum * 4, name = 'conv3', training = training)
            x = dis_conv(x, cnum * 8, name = 'conv4', training = training)
            x = flatten(x, name = 'flatten')
            return x

    def build_wgan_global_discriminator(self, x, reuse = False, training = True):
        with tf.variable_scope('discriminator_global', reuse = reuse):
            cnum = 64
            x = dis_conv(x, cnum, name = 'conv1', training = training)
            x = dis_conv(x, cnum * 2, name = 'conv2', training = training)
            x = dis_conv(x, cnum * 4, name = 'conv3', training = training)
            x = dis_conv(x, cnum * 4, name = 'conv4', training = training)
            x = flatten(x, name = 'flatten')
            return x

    def build_extractor(self, x, reuse = False, training = True):
        with tf.variable_scope('build_extractor', reuse = reuse):
            cnum = 64
            w = x.get_shape()[1]
            x = e_conv(x, cnum, name = 'conv1', training = training)
            x = e_conv(x, cnum * 2, name = 'conv2', training = training)
            x = e_conv(x, cnum * 4, name = 'conv3', training = training)
            x = e_conv(x, cnum * 4, name = 'conv4', training = training)
            x = flatten(x, name = 'flatten')
            z = tf.layers.dense(x, w * w, name = 'z')
            z_var = tf.layers.dense(x, w * w, name = 'z_var')
            return z, z_var

    def build_wgan_discriminator(self, batch_local, batch_global,
                                 reuse = False, training = True):
        with tf.variable_scope('discriminator', reuse = reuse):
            dlocal = self.build_wgan_local_discriminator(
                batch_local, reuse = reuse, training = training)
            dglobal = self.build_wgan_global_discriminator(
                batch_global, reuse = reuse, training = training)
            dout_local = tf.layers.dense(dlocal, 1, name = 'dout_local_fc')
            dout_global = tf.layers.dense(dglobal, 1, name = 'dout_global_fc')

            return dout_local, dout_global

    def build_encode_z(self, batch_local):
        mu, logvar = self.build_extractor(batch_local, reuse = tf.AUTO_REUSE, training = True)

        eps = tf.random_normal(shape = tf.shape(mu))
        std = mu + tf.exp(logvar / 2) * eps

        z = tf.add(mu, tf.multiply(std, eps))

        return mu, logvar, z

    def build_graph_with_losses(self, batch_data, config, training = True,
                                summary = True, reuse = False):
        batch_pos = batch_data / 127.5 - 1.
        # generate mask, 1 represents masked point
#         bbox = random_bbox_np(config)
#         bbox = (32, 32, 64, 64)
        bbox = (tf.constant(config.HEIGHT // 2), tf.constant(config.WIDTH // 2),
                tf.constant(config.HEIGHT), tf.constant(config.WIDTH))
        mask = bbox2mask(bbox, config, name = 'mask_c')  # masked:1
        batch_incomplete = batch_pos * (1. - mask)

        z = tf.random_normal(shape = [batch_incomplete.get_shape()[0].value, batch_incomplete.get_shape()[1].value, batch_incomplete.get_shape()[2].value, 1])
        x1, x2, _ = self.build_inpaint_net(
            batch_incomplete, mask, z, config, reuse = reuse, training = training,
            padding = config.PADDING)
        if config.PRETRAIN_COARSE_NETWORK:
            batch_predicted = x1
            logger.info('Set batch_predicted to x1.')
        else:
            batch_predicted = x2
            logger.info('Set batch_predicted to x2.')
        losses = {}
        # apply mask and complete image(2, 256, 256, 3)
        batch_complete = batch_predicted * mask + batch_incomplete * (1. - mask)
        # local patches(2, 128, 128, 3)
        local_patch_batch_pos = local_patch(batch_pos, bbox)
        local_patch_x1 = local_patch(x1, bbox)
        local_patch_x2 = local_patch(x2, bbox)
        local_patch_batch_complete = local_patch(batch_complete, bbox)
        local_patch_mask = local_patch(mask, bbox)
        l1_alpha = config.COARSE_L1_ALPHA
#         losses['l1_loss'] = l1_alpha * tf.reduce_mean(tf.abs(local_patch_batch_pos - local_patch_x1) * spatial_discounting_mask(config))
#         if not config.PRETRAIN_COARSE_NETWORK:
#             losses['l1_loss'] += tf.reduce_mean(tf.abs(local_patch_batch_pos - local_patch_x2) * spatial_discounting_mask(config))
        losses['ae_loss'] = l1_alpha * tf.reduce_mean(tf.abs(batch_pos - x1) * (1. - mask))
        if not config.PRETRAIN_COARSE_NETWORK:
            losses['ae_loss'] += tf.reduce_mean(tf.abs(batch_pos - x2) * (1. - mask))
        losses['ae_loss'] /= tf.reduce_mean(1. - mask)
        if summary:
#             scalar_summary('losses/l1_loss', losses['l1_loss'])
            scalar_summary('losses/ae_loss', losses['ae_loss'])
            viz_img = [batch_pos, batch_incomplete, batch_complete, x2, x1]
#             if offset_flow is not None:
#                 viz_img.append(
#                     resize(offset_flow, scale = 4,
#                            func = tf.image.resize_nearest_neighbor))
            images_summary(
                tf.concat(viz_img, axis = 2),
                'raw_incomplete_predicted_complete', config.VIZ_MAX_OUT)

        # gan
        batch_pos_neg = tf.concat([batch_pos, batch_complete], axis = 0)
        # local deterministic patch
        local_patch_batch_pos_neg = tf.concat([local_patch_batch_pos, local_patch_batch_complete], 0)
        if config.GAN_WITH_MASK:
            batch_pos_neg = tf.concat([batch_pos_neg, tf.tile(mask, [config.BATCH_SIZE * 2, 1, 1, 1])], axis = 3)
        # wgan with gradient penalty
        if config.GAN == 'wgan_gp':
            # seperate gan
            pos_neg_local, pos_neg_global = self.build_wgan_discriminator(local_patch_batch_pos_neg, batch_pos_neg, training = training, reuse = reuse)
            pos_local, neg_local = tf.split(pos_neg_local, 2)
            pos_global, neg_global = tf.split(pos_neg_global, 2)
            # regress z loss
            mu, logvar, z_encode = self.build_encode_z(local_patch_batch_pos_neg)
            mu_real, mu_fake = tf.split(mu, 2)
            logvar_real, _ = tf.split(logvar, 2)
            z_real, _ = tf.split(z_encode, 2)
            local_patch_z = local_patch(z, bbox)
            z_real = tf.reshape(z_real, local_patch_z.get_shape())
            padding = [[0, 0], [bbox[0], bbox[0]], [bbox[1], bbox[1]], [0, 0]]
#             padding = [[0, 0], [32, 32], [32, 32], [0, 0]]
            z_real = tf.pad(z_real, padding)
            z_fake_label = flatten(local_patch_z, 'z_fake_label')
            losses['l1_regress_z_loss'] = config.REGRESSION_Z_LOSS_ALPHA * tf.reduce_mean(tf.abs(mu_fake - z_fake_label))
            scalar_summary('losses/l1_regress_z_loss', losses['l1_regress_z_loss'])
            z_real_to_x1, z_real_to_x2, _ = self.build_inpaint_net(
                batch_incomplete, mask, z_real, config, reuse = tf.AUTO_REUSE, training = training,
                padding = config.PADDING)

            z_real_batch_predicted = z_real_to_x2
            # apply mask and complete image(2, 256, 256, 3)
            z_real_batch_complete = z_real_batch_predicted * mask + batch_incomplete * (1. - mask)
            # local patches(2, 128, 128, 3)
            z_real_local_patch_x1 = local_patch(z_real_to_x1, bbox)
            z_real_local_patch_x2 = local_patch(z_real_to_x2, bbox)
            losses['l1_loss_z_x'] = l1_alpha * tf.reduce_mean(tf.abs(local_patch_batch_pos - z_real_local_patch_x1))
            if not config.PRETRAIN_COARSE_NETWORK:
                losses['l1_loss_z_x'] += l1_alpha * tf.reduce_mean(tf.abs(local_patch_batch_pos - z_real_local_patch_x2))
            scalar_summary('losses/l1_loss_z_x', losses['l1_loss_z_x'])
            losses['ae_loss_z_x'] = l1_alpha * tf.reduce_mean(tf.abs(batch_pos - z_real_to_x1) * (1. - mask))
            if not config.PRETRAIN_COARSE_NETWORK:
                losses['ae_loss_z_x'] += tf.reduce_mean(tf.abs(batch_pos - z_real_to_x2) * (1. - mask))
            losses['ae_loss_z_x'] /= tf.reduce_mean(1. - mask)
            scalar_summary('losses/ae_loss_z_x', losses['ae_loss_z_x'])
            losses['loss_kl'] = -0.5 * tf.reduce_sum(1 + logvar_real - tf.square(mu_real) - tf.exp(logvar_real))
            scalar_summary('losses/loss_kl', losses['loss_kl'])

            viz_img_res = [batch_pos, batch_incomplete, z_real_batch_complete, z_real_batch_predicted]
            images_summary(
                tf.concat(viz_img_res, axis = 2),
                'res', config.VIZ_MAX_OUT)

            # wgan loss
            g_loss_local, d_loss_local = gan_wgan_loss(pos_local, neg_local, name = 'gan/local_gan')
            g_loss_global, d_loss_global = gan_wgan_loss(pos_global, neg_global, name = 'gan/global_gan')
            losses['g_loss'] = config.GLOBAL_WGAN_LOSS_ALPHA * g_loss_global + g_loss_local
            losses['d_loss'] = d_loss_global + d_loss_local
            # gp
            interpolates_local = random_interpolates(local_patch_batch_pos, local_patch_batch_complete)
            interpolates_global = random_interpolates(batch_pos, batch_complete)
            dout_local, dout_global = self.build_wgan_discriminator(
                interpolates_local, interpolates_global, reuse = True)
            # apply penalty
            penalty_local = gradients_penalty(interpolates_local, dout_local, mask = local_patch_mask)
            penalty_global = gradients_penalty(interpolates_global, dout_global, mask = mask)
            losses['gp_loss'] = config.WGAN_GP_LAMBDA * (penalty_local + penalty_global)
            losses['d_loss'] = losses['d_loss'] + losses['gp_loss']
            if summary and not config.PRETRAIN_COARSE_NETWORK:
                gradients_summary(g_loss_local, batch_predicted, name = 'g_loss_local')
                gradients_summary(g_loss_global, batch_predicted, name = 'g_loss_global')
                scalar_summary('convergence/d_loss', losses['d_loss'])
                scalar_summary('convergence/local_d_loss', d_loss_local)
                scalar_summary('convergence/global_d_loss', d_loss_global)
                scalar_summary('gan_wgan_loss/gp_loss', losses['gp_loss'])
                scalar_summary('gan_wgan_loss/gp_penalty_local', penalty_local)
                scalar_summary('gan_wgan_loss/gp_penalty_global', penalty_global)

        if summary and not config.PRETRAIN_COARSE_NETWORK:
            # summary the magnitude of gradients from different losses w.r.t. predicted image
            gradients_summary(losses['g_loss'], batch_predicted, name = 'g_loss')
            gradients_summary(losses['g_loss'], x1, name = 'g_loss_to_x1')
            gradients_summary(losses['g_loss'], x2, name = 'g_loss_to_x2')
            #gradients_summary(losses['l1_loss'], x1, name = 'l1_loss_to_x1')
            #gradients_summary(losses['l1_loss'], x2, name = 'l1_loss_to_x2')
            gradients_summary(losses['ae_loss'], x1, name = 'ae_loss_to_x1')
            gradients_summary(losses['ae_loss'], x2, name = 'ae_loss_to_x2')
        if config.PRETRAIN_COARSE_NETWORK:
            losses['g_loss'] = 0
        else:
            losses['g_loss'] = config.GAN_LOSS_ALPHA * losses['g_loss']
#         losses['g_loss'] += config.L1_LOSS_ALPHA * losses['l1_loss']
        losses['g_loss'] += config.LOSS_KL * losses['loss_kl']
        losses['g_loss'] += config.REGRESSION_Z_LOSS_ALPHA * losses['l1_loss_z_x']
        losses['g_loss'] += config.REGRESSION_Z_LOSS_ALPHA * losses['ae_loss_z_x']
        logger.info('Set L1_LOSS_ALPHA to %f' % config.L1_LOSS_ALPHA)
        logger.info('Set GAN_LOSS_ALPHA to %f' % config.GAN_LOSS_ALPHA)
        if config.AE_LOSS:
            losses['g_loss'] += config.AE_LOSS_ALPHA * losses['ae_loss']
            logger.info('Set AE_LOSS_ALPHA to %f' % config.AE_LOSS_ALPHA)
        if config.REGRESSION_Z_LOSS:
            losses['g_loss'] += config.REGRESSION_Z_LOSS_ALPHA * losses['l1_regress_z_loss']
            logger.info('Set REGRESSION_Z_LOSS_ALPHA to %f' % config.REGRESSION_Z_LOSS_ALPHA)
        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'inpaint_net')
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        return g_vars, d_vars, losses

    def build_infer_graph(self, batch_data, config, bbox = None, name = 'val'):
        """
        """
        config.MAX_DELTA_HEIGHT = 0
        config.MAX_DELTA_WIDTH = 0
        if bbox is None:
#           bbox = random_bbox_np(config)
          bbox = (tf.constant(config.HEIGHT // 2), tf.constant(config.WIDTH // 2),
                  tf.constant(config.HEIGHT), tf.constant(config.WIDTH))
        mask = bbox2mask(bbox, config, name = name + 'mask_c')
        batch_pos = batch_data / 127.5 - 1.
        batch_incomplete = batch_pos * (1. - mask)
        # inpaint
        z = tf.random_normal(shape = [batch_incomplete.get_shape()[0].value, batch_incomplete.get_shape()[1].value, batch_incomplete.get_shape()[2].value, 1])
        x1, x2, offset_flow = self.build_inpaint_net(
            batch_incomplete, mask, z, config, reuse = True,
            training = False, padding = config.PADDING)
        if config.PRETRAIN_COARSE_NETWORK:
            batch_predicted = x1
            logger.info('Set batch_predicted to x1.')
        else:
            batch_predicted = x2
            logger.info('Set batch_predicted to x2.')
        # apply mask and reconstruct
        batch_complete = batch_predicted * mask + batch_incomplete * (1. - mask)
        # global image visualization
        viz_img = [batch_pos, batch_incomplete, batch_complete]
        if offset_flow is not None:
            viz_img.append(
                resize(offset_flow, scale = 4,
                       func = tf.image.resize_nearest_neighbor))
        images_summary(
            tf.concat(viz_img, axis = 2),
            name + '_raw_incomplete_complete', config.VIZ_MAX_OUT)
        return batch_complete

    def build_static_infer_graph(self, batch_data, config, name):
        """
        """
        # generate mask, 1 represents masked point
        bbox = (tf.constant(config.HEIGHT // 2), tf.constant(config.WIDTH // 2),
                tf.constant(config.HEIGHT), tf.constant(config.WIDTH))
        return self.build_infer_graph(batch_data, config, bbox, name)

    def build_server_graph(self, batch_data, reuse = False, is_training = False):
        """
        """
        # generate mask, 1 represents masked point
        batch_raw, masks_raw = tf.split(batch_data, 2, axis = 2)
        masks = tf.cast(masks_raw[0:1, :, :, 0:1] > 127.5, tf.float32)

        batch_pos = batch_raw / 127.5 - 1.
        batch_incomplete = batch_pos * (1. - masks)
        # inpaint
        z = tf.random_normal(shape = [batch_incomplete.get_shape()[0].value, batch_incomplete.get_shape()[1].value, batch_incomplete.get_shape()[2].value, 1])
        _, x2, _ = self.build_inpaint_net(
            batch_incomplete, masks, z, reuse = reuse, training = is_training,
            config = None)
        batch_predict = x2
        # apply mask and reconstruct
        batch_complete = batch_predict * masks + batch_incomplete * (1 - masks)
        return batch_complete

    def build_server_graph__(self, batch_data, reuse = False, is_training = False):
        """
        """
        # generate mask, 1 represents masked point
        batch_raw, masks_raw = tf.split(batch_data, 2, axis=2)
        masks =  tf.cast(masks_raw[0:1, :, :, 0:1] > 127.5, tf.float32)

        bbox = (tf.constant(64 // 2), tf.constant(64 // 2),
        tf.constant(64), tf.constant(64))

        batch_pos = batch_raw / 127.5 - 1.
        batch_incomplete = batch_pos * (1. - masks)
        #inpaint
        local_patch_batch_pos = local_patch(batch_pos, bbox)
        mu, logvar, z_encode = self.build_encode_z(local_patch_batch_pos)

        z_encode =  tf.reshape(z_encode, (34, 64, 64, 1))
        padding = [[0, 0], [32, 32], [52, 32], [0, 0]]
        z_encode = tf.pad(z_encode, padding)

        return z_encode
